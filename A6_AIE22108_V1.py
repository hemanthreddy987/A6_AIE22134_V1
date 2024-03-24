# Defining the perceptron function with "step activation function"

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
#A1
# Perceptron function
def perceptron_func(X, W, learning_rate, epochs): # Function definition
    errors = []
    epoch = 0
    for epoch in range(epochs):
        error = 0 # Initializes the error for the current epoch
        for i in range(len(X)): # For each epoch, it iterates through the input data (X) and updates the weights
            y = np.dot(X[i, :-1], W[1:]) + W[0] # Calculate the dot product of inputs and corresponding weights, plus bias
            output = 1 if y > 0 else 0 # Determines the output of the perceptron based on the threshold
            error += (0.5 * (X[i][-1] - output) ** 2) # Calculates the squared error between the target output (X[i][-1]) and the perceptron's output (output),
                                                      # adding it to the total error for the epoch
            W[1:] += learning_rate * (X[i][-1] - output) * X[i, :-1] # Updates weights for input features
            W[0] += learning_rate * (X[i][-1] - output) # Updates the bias
        errors.append(error)  # Stores the total error for the current epoch
        if error <= 0.002:  # Check if the error is below the specified threshold
            break
    print ("The number of epochs required for weights to converge (Step Activation Function) =", epoch)
    print ("\n\nEpoch vs Error graph -")
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Epochs vs Error')
    plt.show()
    
    # Implementation of AND gate logic on Perceptron Function (Step Activation Function)

# AND gate truth table defined in an array
X = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])
# Initialising the Weights of W0, W1 and W2
W = np.array([10, 0.2, -0.75])

# Initialising the maximum learning rate and epochs
learning_rate = 0.05
epochs = 1000

# Call the perceptron function for the AND gate
perceptron_func(X, W, learning_rate, epochs)

#A2
# Defining the perceptron function with "Bi-Polar, Sigmoid and ReLU functions"

import numpy as np
import matplotlib.pyplot as plt

# Perceptron function with different activation functions
def perceptron_with_activation(X, W, learning_rate, epochs, activation_function):
    errors = []  # Initialize a list to store errors for each epoch
    for epoch in range(epochs):  # Iterate through the training process for the specified number of epochs
        error = 0  # Initialize the error for the current epoch
        for i in range(len(X)):  # Iterate through each data point in the input data X
            y = np.dot(X[i, :-1], W[1:]) + W[0]  # Calculate the dot product of inputs and weights, plus bias
            if activation_function == "bipolar_step":
                output = 1 if y > 0 else -1  # Bi-Polar Step function
            elif activation_function == "sigmoid":
                output = 1 / (1 + np.exp(-y))  # Sigmoid function
            elif activation_function == "relu":
                output = max(0, y)  # ReLU function
            error += (0.5 * (X[i][-1] - output) ** 2)  # Calculate the error for the current data point
            W[1:] += learning_rate * (X[i][-1] - output) * X[i, :-1]  # Update weights for input features
            W[0] += learning_rate * (X[i][-1] - output)  # Update the bias (weight for the constant input)
        errors.append(error)  # Store the total error for the current epoch
        if error <= 0.002:  # Check if the error is below the specified threshold
            break  # If so, break out of the training loop
    # Print the number of epochs required for weights to converge and plot the Epochs vs Error graph
    print("\n\nThe number of epochs required for weights to converge(", activation_function, ")=", epoch)
    print("Epoch vs Error graph -")
    plt.plot(range(1, len(errors) + 1), errors, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('Epochs vs Error (' + activation_function + ' activation)')
    plt.show()
    
    # Implementation of AND gate logic on Perceptron Function (With Bi-Polar, Sigmoid and ReLU functions)

# AND gate truth table defined in an array
X = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

# Initial weights
W = np.array([10, 0.2, -0.75])

# Initialising the maximum learning rate and epochs
learning_rate = 0.05
epochs = 1000

# Call the perceptron function with different activation functions

perceptron_with_activation(X, W, learning_rate, epochs, "bipolar_step")  # Bi-Polar Step function
perceptron_with_activation(X, W, learning_rate, epochs, "sigmoid")  # Sigmoid function
perceptron_with_activation(X, W, learning_rate, epochs, "relu")  # ReLU function


#A3
# Implementation of Perceptron Function (Step Activation Function) with different Learning Rates
import numpy as np
import matplotlib.pyplot as plt

# Perceptron function
def perceptron_func_diffLearningRates(X, W, learning_rate, epochs):
    errors = []
    for epoch in range(epochs):
        error = 0
        for i in range(len(X)):
            y = np.dot(X[i, :-1], W[1:]) + W[0]
            output = 1 if y > 0 else 0
            error += (0.5 * (X[i][-1] - output) ** 2)
            W[1:] += learning_rate * (X[i][-1] - output) * X[i, :-1]
            W[0] += learning_rate * (X[i][-1] - output)
        errors.append(error)
        if error <= 0.002:
            break
    return epoch


# Implementation of AND gate logic on Perceptron Function (Step Activation Function) with different Learning Rates
# AND gate truth table defined in an array
X = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
    [1, 1, 1]
])

# Initialising the Weights of W0, W1, and W2
W = np.array([10, 0.2, -0.75])

# Initialising the learning rates and epochs
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
epochs = 1000

# Calculate the number of iterations for each learning rate
iterations = []
for lr in learning_rates:
    W_temp = W.copy()
    iterations.append(perceptron_func_diffLearningRates(X, W_temp, lr, epochs))
    print(f'Learning Rate: {lr}, Number of Iterations: {iterations}')

# Plotting the results
plt.plot(learning_rates, iterations, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations for Convergence')
plt.title('Learning Rate vs Iterations')
plt.show()


# A4 (A1)
# Implementation of XOR gate logic on Perceptron Function (Step Activation Function)

# XOR gate truth table defined in an array
X = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])
# Initialising the Weights of W0, W1 and W2
W = np.array([10, 0.2, -0.75])

# Initialising the maximum learning rate and epochs
learning_rate = 0.05
epochs = 1000

# Call the perceptron function for the XOR gate
perceptron_func(X, W, learning_rate, epochs)


#A4 (A2)
# Implementation of XOR gate logic on Perceptron Function (With Bi-Polar, Sigmoid and ReLU functions)

# AND gate truth table defined in an array
X = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])
# Initial weights
W = np.array([10, 0.2, -0.75])

# Initialising the maximum learning rate and epochs
learning_rate = 0.05
epochs = 1000

# Call the perceptron function with different activation functions
perceptron_with_activation(X, W, learning_rate, epochs, "bipolar_step")  # Bi-Polar Step function
perceptron_with_activation(X, W, learning_rate, epochs, "sigmoid")  # Sigmoid function
perceptron_with_activation(X, W, learning_rate, epochs, "relu")  # ReLU function


# A4 (A3)
# Implementation of XOR gate logic on Perceptron Function (Step Activation Function) with different Learning Rates
# AND gate truth table defined in an array
X = np.array([
    [0, 0, 0],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 0]
])

# Initialising the Weights of W0, W1, and W2
W = np.array([10, 0.2, -0.75])

# Initialising the learning rates and epochs
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
epochs = 1000

# Calculate the number of iterations for each learning rate
iterations = []
for lr in learning_rates:
    W_temp = W.copy()
    iterations.append(perceptron_func_diffLearningRates(X, W_temp, lr, epochs))
    print(f'Learning Rate: {lr}, Number of Iterations: {iterations}')

# Plotting the results
plt.plot(learning_rates, iterations, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Number of Iterations for Convergence')
plt.title('Learning Rate vs Iterations')
plt.show()



import numpy as np
#A5
# Given data
data = np.array([
    [1, 20, 6, 2, 386, 1],
    [2, 16, 3, 6, 289, 1],
    [3, 27, 6, 2, 393, 1],
    [4, 19, 1, 2, 110, 0],
    [5, 24, 4, 2, 280, 1],
    [6, 22, 1, 5, 167, 0],
    [7, 15, 4, 2, 271, 1],
    [8, 18, 4, 2, 274, 1],
    [9, 21, 1, 4, 148, 0],
    [10, 16, 2,4, 198, 0]
])

# Data preprocessing
X = data[:, 1:-1]  # Features (Candies, Mangoes, Milk Packets, Payment)
y = data[:, -1].reshape(-1, 1)  # Labels (High Value Tx?)

# Normalizing the features
X_normalized = X / np.max(X, axis=0)

# Initializing weights and bias
np.random.seed(42)
weights = np.random.rand(X.shape[1], 1)
bias = np.random.rand(1, 1)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Setting learning rate
learning_rate = 0.01

# Training the perceptron
for epoch in range(10000):
    # Forward pass
    weighted_sum = np.dot(X_normalized, weights) + bias
    activation = sigmoid(weighted_sum)

    # Calculating error
    error = y - activation

    # Backpropagation
    d_weights = learning_rate * np.dot(X_normalized.T, error * sigmoid_derivative(activation))
    d_bias = learning_rate * np.sum(error * sigmoid_derivative(activation))

    # Updating weights and bias
    weights += d_weights
    bias += d_bias

# Testing the perceptron
test_data = np.array([[11, 14, 21, 48]])
test_data_normalized = test_data / np.max(X, axis=0)
prediction = sigmoid(np.dot(test_data_normalized, weights) + bias)

# Classifying as High or Low Value
result = "High Value" if prediction >= 0.5 else "Low Value"

print(f"Prediction: {prediction}, Classification: {result}")



import numpy as np
#A6
# Given data
data = np.array([
    [1, 20, 6, 2, 386, 1],
    [2, 16, 3, 6, 289, 1],
    [3, 27, 6, 2, 393, 1],
    [4, 19, 1, 2, 110, 0],
    [5, 24, 4, 2, 280, 1],
    [6, 22, 1, 5, 167, 0],
    [7, 15, 4, 2, 271, 1],
    [8, 18, 4, 2, 274, 1],
    [9, 21, 1, 4, 148, 0],
    [10, 16, 2, 4, 198, 0]
])

# Data preprocessing
X = data[:, 1:-1]  # Features (Candies, Mangoes, Milk Packets, Payment)
y = data[:, -1].reshape(-1, 1)  # Labels (High Value Tx?)

# Normalizing the features
X_normalized = X / np.max(X, axis=0)

# Add a bias term to the features
X_bias = np.hstack([X_normalized, np.ones((X_normalized.shape[0], 1))])

# Initialize weights and bias for perceptron
np.random.seed(42)
weights_perceptron = np.random.rand(X_bias.shape[1], 1)
bias_perceptron = np.random.rand(1, 1)

# Calculate pseudo-inverse weights 
weights_pseudo_inverse = np.linalg.pinv(X_bias) @ y

# Testing the perceptron
test_data = np.array([[11, 14, 21, 48]])
test_data_normalized = test_data / np.max(X, axis=0)
test_data_bias = np.hstack([test_data_normalized, np.ones((1, 1))])
prediction_perceptron = sigmoid(np.dot(test_data_bias, weights_perceptron) + bias_perceptron)
prediction_pseudo_inverse = sigmoid(test_data_bias @ weights_pseudo_inverse)

# Classify as High or Low Value
result_perceptron = "High Value" if prediction_perceptron >= 0.5 else "Low Value"
result_pseudo_inverse = "High Value" if prediction_pseudo_inverse >= 0.5 else "Low Value"

# Compare results
print(f"Perceptron Prediction: {prediction_perceptron}, Classification: {result_perceptron}")
print(f"Pseudo-inverse Prediction: {prediction_pseudo_inverse}, Classification: {result_pseudo_inverse}")



# Neural Network Model without Bias for AND Gate with 1 output node
import numpy as np
#A7
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

# Initialize weights without biases
np.random.seed(1)
input_hidden_weights = np.random.rand(2, 2)
hidden_output_weights = np.random.rand(2, 1)

# Learning rate and convergence error
learning_rate = 0.05
convergence_error = 0.002

# Training the neural network without biases
for epoch in range(1000):
    # Forward pass
    hidden_layer_input = np.dot(X, input_hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, hidden_output_weights)
    predicted_output = sigmoid(output_layer_input)

    # Calculation of error
    error = y - predicted_output

    # Backpropagation
    output_error = error * sigmoid_derivative(predicted_output)
    hidden_error = output_error.dot(hidden_output_weights.T) * sigmoid_derivative(hidden_layer_output)

    # Updating weights
    hidden_output_weights += hidden_layer_output.T.dot(output_error) * learning_rate
    input_hidden_weights += X.T.dot(hidden_error) * learning_rate

    # Checking for convergence
    if np.mean(np.abs(error)) <= convergence_error:
        print(f"Converged at epoch {epoch}")
        break

# Test the trained neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_hidden = sigmoid(np.dot(test_data, input_hidden_weights))
test_output = sigmoid(np.dot(test_hidden, hidden_output_weights))
print("Predictions:")
for i in range(len(test_data)):
    print(f"Input: {test_data[i]}, Predicted Output: {test_output[i]}")
    
    
    # Neural Network Model with Bias for AND Gate with 1 output node
# The code works when it runs for 10,000 epochs instead of 1000 epochs
import numpy as np

# Define the AND gate input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [0],
              [0],
              [1]])

# Set the learning rate and initialize weights and bias
learning_rate = 0.05
input_size = X.shape[1]
hidden_size = 2
output_size = 1

# Initialize weights and bias with random values
np.random.seed(42)
V = np.random.rand(input_size, hidden_size)
W = np.random.rand(hidden_size, output_size)
bias_hidden = np.random.rand(1, hidden_size)
bias_output = np.random.rand(1, 1)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network using backpropagation with convergence condition
epochs = 1000
convergence_error = 0.002

for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, V) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, W) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Calculate error
    error = y - predicted_output

    if np.mean(np.abs(error)) <= convergence_error:
        print(f"Converged at epoch {epoch}")
        break

    # Backpropagation
    delta_output = error * sigmoid_derivative(predicted_output)
    error_hidden = delta_output.dot(W.T)
    delta_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and bias
    W += learning_rate * hidden_layer_output.T.dot(delta_output)
    bias_output += learning_rate * np.sum(delta_output, axis=0, keepdims=True)

    V += learning_rate * X.T.dot(delta_hidden)
    bias_hidden += learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

# Test the neural network and display results
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
hidden_layer_input_test = np.dot(test_input, V) + bias_hidden
hidden_layer_output_test = sigmoid(hidden_layer_input_test)

output_layer_input_test = np.dot(hidden_layer_output_test, W) + bias_output
predicted_output_test = sigmoid(output_layer_input_test)

# Classify predicted output as 0 or 1
predicted_output_binary = np.round(predicted_output_test)

# Display the results
for i in range(len(test_input)):
    print(f"Input: {test_input[i]}, Predicted Output: {predicted_output_binary[i]}")
    
    
    
    
    # Neural Network Model without Bias (as given in the architecture) for XOR Gate with 1 output node
import numpy as np
#A8
# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input data for XOR gate
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output data for XOR gate
y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights without biases
np.random.seed(1)
input_hidden_weights = np.random.rand(2, 2)
hidden_output_weights = np.random.rand(2, 1)

# Learning rate and convergence error
learning_rate = 0.05
convergence_error = 0.002

# Train the neural network without biases
for epoch in range(1000):
    # Forward pass
    hidden_layer_input = np.dot(X, input_hidden_weights)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, hidden_output_weights)
    predicted_output = sigmoid(output_layer_input)

    # Calculate error
    error = y - predicted_output

    # Backpropagation
    output_error = error * sigmoid_derivative(predicted_output)
    hidden_error = output_error.dot(hidden_output_weights.T) * sigmoid_derivative(hidden_layer_output)

    # Update weights
    hidden_output_weights += hidden_layer_output.T.dot(output_error) * learning_rate
    input_hidden_weights += X.T.dot(hidden_error) * learning_rate

    # Check for convergence
    if np.mean(np.abs(error)) <= convergence_error:
        print(f"Converged at epoch {epoch}")
        break

# Test the trained neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_hidden = sigmoid(np.dot(test_data, input_hidden_weights))
test_output = sigmoid(np.dot(test_hidden, hidden_output_weights))
print("Predictions:")
for i in range(len(test_data)):
    print(f"Input: {test_data[i]}, Predicted Output: {test_output[i]}")
    
    
    # Neural Network Model with Bias for XOR Gate with 1 output node
# The code works when it runs for 10,000 epochs instead of 1000 epochs
import numpy as np

# Define the XOR gate input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Set the learning rate and initialize weights and bias
learning_rate = 0.05
input_size = X.shape[1]
hidden_size = 2
output_size = 1

# Initialize weights and bias with random values
np.random.seed(42)
V = np.random.rand(input_size, hidden_size)
W = np.random.rand(hidden_size, output_size)
bias_hidden = np.random.rand(1, hidden_size)
bias_output = np.random.rand(1, 1)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network using backpropagation with convergence condition
epochs = 1000
convergence_error = 0.002

for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, V) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, W) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Calculate error
    error = y - predicted_output

    if np.mean(np.abs(error)) <= convergence_error:
        print(f"Converged at epoch {epoch}")
        break

    # Backpropagation
    delta_output = error * sigmoid_derivative(predicted_output)
    error_hidden = delta_output.dot(W.T)
    delta_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and bias
    W += learning_rate * hidden_layer_output.T.dot(delta_output)
    bias_output += learning_rate * np.sum(delta_output, axis=0, keepdims=True)

    V += learning_rate * X.T.dot(delta_hidden)
    bias_hidden += learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

# Test the neural network and display results
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
hidden_layer_input_test = np.dot(test_input, V) + bias_hidden
hidden_layer_output_test = sigmoid(hidden_layer_input_test)

output_layer_input_test = np.dot(hidden_layer_output_test, W) + bias_output
predicted_output_test = sigmoid(output_layer_input_test)

# Classify predicted output as 0 or 1
predicted_output_binary = np.round(predicted_output_test)

# Display the results
for i in range(len(test_input)):
    print(f"Input: {test_input[i]}, Predicted Output: {predicted_output_binary[i]}")
    
    
    
    # Neural Network Model with Bias for XOR Gate with 1 output node
# The code works when it runs for 10,000 epochs instead of 1000 epochs
import numpy as np

# Define the XOR gate input and output
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Set the learning rate and initialize weights and bias
learning_rate = 0.05
input_size = X.shape[1]
hidden_size = 2
output_size = 1

# Initialize weights and bias with random values
np.random.seed(42)
V = np.random.rand(input_size, hidden_size)
W = np.random.rand(hidden_size, output_size)
bias_hidden = np.random.rand(1, hidden_size)
bias_output = np.random.rand(1, 1)

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network using backpropagation with convergence condition
epochs = 1000
convergence_error = 0.002

for epoch in range(epochs):
    # Forward pass
    hidden_layer_input = np.dot(X, V) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, W) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Calculate error
    error = y - predicted_output

    if np.mean(np.abs(error)) <= convergence_error:
        print(f"Converged at epoch {epoch}")
        break

    # Backpropagation
    delta_output = error * sigmoid_derivative(predicted_output)
    error_hidden = delta_output.dot(W.T)
    delta_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and bias
    W += learning_rate * hidden_layer_output.T.dot(delta_output)
    bias_output += learning_rate * np.sum(delta_output, axis=0, keepdims=True)

    V += learning_rate * X.T.dot(delta_hidden)
    bias_hidden += learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

# Test the neural network and display results
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
hidden_layer_input_test = np.dot(test_input, V) + bias_hidden
hidden_layer_output_test = sigmoid(hidden_layer_input_test)

output_layer_input_test = np.dot(hidden_layer_output_test, W) + bias_output
predicted_output_test = sigmoid(output_layer_input_test)

# Classify predicted output as 0 or 1
predicted_output_binary = np.round(predicted_output_test)

# Display the results
for i in range(len(test_input)):
    print(f"Input: {test_input[i]}, Predicted Output: {predicted_output_binary[i]}")
    
    
    
    # A9 (A7)
# Neural Network Model without Bias (as given in the architecture) for AND Gate with 2 output node
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output dataset
y = np.array([[1, 0],
              [1, 0],
              [1, 0],
              [0, 1]])

# Initialize weights
np.random.seed(1)
input_nodes = 2
hidden_nodes = 2
output_nodes = 2

weights_input_hidden = np.random.uniform(size=(input_nodes, hidden_nodes))
weights_hidden_output = np.random.uniform(size=(hidden_nodes, output_nodes))

# Training the neural network
learning_rate = 0.05
epochs = 1000

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

    # Calculate mean squared error
    mse = np.mean(np.square(error))

    if mse <= 0.002:
        print(f"Converged in {epoch + 1} epochs with MSE: {mse}")
        break

if mse > 0.002:
    print("Learning did not converge within 1000 epochs.")

# Testing the trained network
test_input = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

hidden_layer_output = sigmoid(np.dot(test_input, weights_input_hidden))
output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output))
print("Final output:")
print(output)


# A9 (A8)
# Neural Network Model without Bias (as given in the architecture) for XOR Gate with 2 output node
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output dataset
y = np.array([[1, 0],
              [0, 1],
              [0, 1],
              [1, 0]])

# Initialize weights
np.random.seed(1)
input_nodes = 2
hidden_nodes = 2
output_nodes = 2

weights_input_hidden = np.random.uniform(size=(input_nodes, hidden_nodes))
weights_hidden_output = np.random.uniform(size=(hidden_nodes, output_nodes))

# Training the neural network
learning_rate = 0.05
epochs = 1000

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

    # Calculate mean squared error
    mse = np.mean(np.square(error))

    if mse <= 0.002:
        print(f"Converged in {epoch + 1} epochs with MSE: {mse}")
        break

if mse > 0.002:
    print("Learning did not converge within 1000 epochs.")

# Testing the trained network
test_input = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

hidden_layer_output = sigmoid(np.dot(test_input, weights_input_hidden))
output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output))
print("Final output:")
print(output)



# A9 (A7)
# Neural Network Model with Bias for AND Gate with 2 output node
# Learning converged in 7918 epochs
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output dataset
y = np.array([[1, 0],
              [1, 0],
              [1, 0],
              [0, 1]])

# Initialize weights and biases
np.random.seed(1)
input_nodes = 2
hidden_nodes = 2
output_nodes = 2

weights_input_hidden = np.random.uniform(size=(input_nodes, hidden_nodes))
bias_hidden = np.random.uniform(size=(1, hidden_nodes))

weights_hidden_output = np.random.uniform(size=(hidden_nodes, output_nodes))
bias_output = np.random.uniform(size=(1, output_nodes))

# Training the neural network
learning_rate = 0.05
epochs = 1000

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Calculate mean squared error
    mse = np.mean(np.square(error))

    if mse <= 0.002:
        print(f"Converged in {epoch + 1} epochs with MSE: {mse}")
        break

if mse > 0.002:
    print("Learning did not converge within 1000 epochs.")

# Testing the trained network
test_input = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

hidden_layer_output = sigmoid(np.dot(test_input, weights_input_hidden) + bias_hidden)
output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output) + bias_output)
print("Final output:")
print(output)


# A9 (A8)
# Neural Network Model with Bias for XOR Gate with 2 output node
# Learning converged in 23859 epochs
import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output dataset
y = np.array([[1, 0],
              [0, 1],
              [0, 1],
              [1, 0]])

# Initialize weights and biases
np.random.seed(1)
input_nodes = 2
hidden_nodes = 2
output_nodes = 2

weights_input_hidden = np.random.uniform(size=(input_nodes, hidden_nodes))
bias_hidden = np.random.uniform(size=(1, hidden_nodes))

weights_hidden_output = np.random.uniform(size=(hidden_nodes, output_nodes))
bias_output = np.random.uniform(size=(1, output_nodes))

# Training the neural network
learning_rate = 0.05
epochs = 1000

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - output
    d_output = error * sigmoid_derivative(output)

    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    bias_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    bias_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Calculate mean squared error
    mse = np.mean(np.square(error))

    if mse <= 0.002:
        print(f"Converged in {epoch + 1} epochs with MSE: {mse}")
        break

if mse > 0.002:
    print("Learning did not converge within 1000 epochs.")

# Testing the trained network
test_input = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

hidden_layer_output = sigmoid(np.dot(test_input, weights_input_hidden) + bias_hidden)
output = sigmoid(np.dot(hidden_layer_output, weights_hidden_output) + bias_output)
print("Final output:")
print(output)



from sklearn.neural_network import MLPClassifier
#A10
# AND Gate dataset
X_and = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_and = [0, 0, 0, 1]

# XOR Gate dataset
X_xor = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_xor = [0, 1, 1, 0]

# Create MLPClassifier models
and_model = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', solver='lbfgs', random_state=1)
xor_model = MLPClassifier(hidden_layer_sizes=(4,), activation='logistic', solver='lbfgs', random_state=1)

# Train the models
and_model.fit(X_and, y_and)
xor_model.fit(X_xor, y_xor)

# Test the models
print("AND Gate Predictions:", and_model.predict(X_and))
print("XOR Gate Predictions:", xor_model.predict(X_xor))

#A11
import os
import cv2
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
# Define the path to the directory containing the training data
train_data_path = r"train"

# Initialize lists to store file paths and labels
filepaths = []
labels = []

# Get the list of subdirectories (class labels)
folds = os.listdir(train_data_path)

# Iterate over each subdirectory
for fold in folds:
    # Get the full path to the subdirectory
    f_path = os.path.join(train_data_path, fold)
    # Get the list of file names in the subdirectory
    filelists = os.listdir(f_path)
    
    # Iterate over each file in the subdirectory
    for file in filelists:
        # Get the full path to the file
        filepaths.append(os.path.join(f_path, file))
        # Store the label (subdirectory name) for the file
        labels.append(fold)

# Initialize a list to store image vectors
images = []

# Iterate over each file path
for filepath in filepaths:
    # Read the image from the file
    img = cv2.imread(filepath)
    # Resize the image to a fixed size
    img = cv2.resize(img, (100, 100))  # Adjust the size as needed
    # Flatten the image into a 1D array
    img_vector = img.flatten()
    # Append the flattened image vector to the list
    images.append(img_vector)

# Convert the list of image vectors to a numpy array
images_array = np.array(images)

# Create a DataFrame to store the image vectors and labels
df = pd.DataFrame(images_array, columns=[f"pixel_{i}" for i in range(images_array.shape[1])])
df['label'] = labels

print("Shape of DataFrame:", df.shape)
print("Head of DataFrame:", df.head())

# Separate data into two classes: "normal" and "OSSC"
normal_class = df[df['label'] == 'Normal']
oscc_class = df[df['label'] == 'OSCC']

print("Shape of Normal class:", normal_class.shape)
print("Shape of OSCC class:", oscc_class.shape)
X = df.drop('label', axis=1)  # Features (pixel_0 to pixel_29999)
y = df['label']  # Class labels
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create and train the MLP Classifier model
mlp_classifier = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', solver='lbfgs', random_state=1,max_iter = 10000)
mlp_classifier.fit(X_train, y_train)

# Make predictions
y_pred = mlp_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)