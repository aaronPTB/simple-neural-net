import numpy as np
from identify import NeuralNetwork

nn = NeuralNetwork(2, 1, 2)

nn.forward([[1, 0]])

# nn.displayLayers()

for i in range(2):
    nn.train(np.array([[1, 0]]),[[.2, .8]])

# nn.displayLayers()
