# Inputs to the layer
inputs = [1, 2, 3, 2.5] 

# Weights for each of the 3 neurons
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

# Biases for each neuron
biases = [2, 3, 0.5]


layer_outputs = []      # Output of current layer

# Iterating through each neuron's weithts and bias

for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0   # Output the given neuron
        
        for n_input, weight in zip(inputs, neuron_weights):
                neuron_output += n_input*weight
                
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)

print(layer_outputs)

