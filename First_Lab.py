# Input matrix of size 3x1
input_matrix = np.array([[0.5], [0.1], [0.2]])

# Weights and biases for the first hidden layer (1 neuron)
W1 = np.array([[0.4, 0.3, 0.2]])  # 1x3 matrix
b1 = np.array([[0.1]])  # Bias for 1 neuron

# Weights and biases for the second case (2 hidden layers, 1 neuron each)
W2 = np.array([[0.5, 0.3, 0.7]])  # 1st hidden layer
b2 = np.array([[0.2]])  # Bias for 1st hidden layer

W3 = np.array([[0.1]])  # 2nd hidden layer (connected to the output of 1st hidden layer)
b3 = np.array([[0.4]])  # Bias for 2nd hidden layer

def matrix_multiply(A, B):
    return np.dot(A, B)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward propagation for 1 hidden layer
Z1 = matrix_multiply(W1, input_matrix) + b1  # Linear step
A1 = sigmoid(Z1)  # Activation step

print("Output from the first hidden layer:", A1)

# Forward propagation for 2 hidden layers
# First hidden layer
Z2 = matrix_multiply(W2, input_matrix) + b2
A2 = sigmoid(Z2)

# Second hidden layer
Z3 = matrix_multiply(W3, A2) + b3
A3 = sigmoid(Z3)

print("Output from the second hidden layer:", A3)