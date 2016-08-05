from identify.py import NeuralNetwork

nn = NeuralNetwork(1, 1, 1)

inp = np.transpose(np.array([1,0]))

nn.forward([1])
nn.displayLayers()

nn.train([1],[0])

np.forward([1])
np.displayLayers()
