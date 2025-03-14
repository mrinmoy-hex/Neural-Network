import numpy as np

# Inputs to the first layer (3 samples, each with 4 features)
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]

# Weights for the first layer (3 neurons, each with 4 input connections)
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# Biases for each neuron in the first layer
biases = [2, 3, 0.5]

# Weights for the second layer (3 neurons, each receiving 3 inputs from layer 1)
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

# Biases for each neuron in the second layer
biases2 = [-1, 2, -0.5]

# Compute the output of the first layer:
# - Matrix multiplication of inputs (3x4) with weights (3x4 transposed to 4x3)
# - Adds biases (broadcasted across each row)
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# Compute the output of the second layer:
# - Uses the outputs from the first layer as inputs for the second layer
# - Multiplies by second layer weights (3x3 transposed to 3x3)
# - Adds biases (broadcasted across each row)
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# Print the final output of the second layer
print(layer2_outputs)
