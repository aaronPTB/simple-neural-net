import numpy as np
from identify import NeuralNetwork

nn = NeuralNetwork(2, 1, 2)

nn.forward([[1, 0, .5]])

nn.displayLayers()

for i in range(20000):
    print nn.cost(np.array([[1, 0, .5]]),np.array([[.2, .8]]))
    nn.train(np.array([[1, 0, .5]]),np.array([[.2, .8]]))

nn.displayLayers()
