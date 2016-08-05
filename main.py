import numpy as np
from identify import NeuralNetwork

nn = NeuralNetwork(2, 1, 1)

nn.forward([[1, 0]])

nn.displayLayers()

for i in range(1):
    print nn.cost(np.array([[1, 0]]),np.array([[.2, .8]]))
    nn.train(np.array()

nn.displayLayers()
