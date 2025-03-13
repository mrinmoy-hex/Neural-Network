import numpy as np

# Inputs to the layer
inputs = [1, 2, 3, 2.5] 

# Weights for each of the 3 neurons
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# Biases for each neuron
biases = [2, 3, 0.5]

output = np.dot(weights, inputs) + biases
print(output)