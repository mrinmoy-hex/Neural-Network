import numpy as np

# Set random seed for reproducibility (ensures same random values every time)
np.random.seed(0)

# Input data: 3 samples (rows) with 4 features (columns) each
X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]

# Define a Dense Layer class
class NeuralNetwork:
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights with small random values (Gaussian distribution)
        # Shape: (n_inputs, n_neurons)
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        
        # Initialize biases with zeros, shape: (1, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    # Forward pass (calculates weighted sum + bias)
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
    
# Create first layer: 4 input features -> 5 neurons
layer1 = NeuralNetwork(4, 5)
# Create second layer: 5 input features (from layer1) -> 2 neurons
layer2 = NeuralNetwork(5, 2)

# Perform forward pass through first layer
layer1.forward(X)
print("Layer 1 Output:\n", layer1.output)

# Perform forward pass through second layer (using output from layer1 as input)
layer2.forward(layer1.output)
print("Layer 2 Output:\n", layer2.output)
