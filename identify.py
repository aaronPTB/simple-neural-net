import numpy as np

class NeuralNetwork:
    """Neural network class that allows for simple foreward and backward
    propogation with a momentum-based stochastic gradient descent method.

    Has a dependency on numpy"""

    def __init__(self, input_size, hidden_size, output_size,
                 learn_rate = .1, reg_lambda=.01):
        print "Neural network initialized with random weights"
        self.weights_ih = np.random.rand(hidden_size, input_size)
        self.weights_ho = np.random.rand(output_size, hidden_size)
        self.bias_ih    = np.random.rand(1, hidden_size).transpose()
        self.bias_ho    = np.random.rand(1, output_size).transpose()
        self.learn_rate = learn_rate
        self.reg_lambda = reg_lambda
        self.input      = np.zeros((1, 2))
        self.hidden_lin = np.zeros((1, hidden_size))
        self.hidden     = np.zeros((1, hidden_size))
        self.output     = np.zeros((1, output_size))

    def forward(self, matrix):
        ## Propegating results foreward and applying a nonlinearity
        self.displayLayers()

        self.input = np.array(matrix).transpose()
        self.hidden_lin = np.dot(self.weights_ih, self.input)
        self.hidden = self.act(self.hidden_lin)
        self.output = np.dot(self.weights_ho, self.hidden)

        return self.output
        ## Softmaxing output
        # self.output = np.exp(self.output)
        # return self.output/np.sum(self.output)
    def train(self, matrix, expected):

        current_cost_prime = self.costPrime(matrix, np.array(expected))

        ## Calculates the layer 3 error
        error_layer_3 = current_cost_prime * self.output
        print error_layer_3.shape
        dJdW_ho = self.hidden.transpose().dot(error_layer_3.transpose())

        ## Calculates the error for layer two by propogating back the error
        ## And applying the chain rule
        error_layer_2 = error_layer_3 * self.weights_ho.transpose() * self.actPrime(self.hidden_lin)
        dJdW_ih = np.transpose(matrix).dot(error_layer_2)

        self.weights_ho = self.weights_ho - dJdW_ho * self.learn_rate
        self.weights_ih = self.weights_ih - dJdW_ih * self.learn_rate

    def act(self, matrix):
        ## Activation function
        return np.tanh(matrix)

    def actPrime(self, matrix):
        ## Derivative of activation function
        return np.square(1/np.cosh(matrix))

    def cost(self, matrix, expected):
        ## Uses cross entropy with a regularization constant added on top to
        ## Calculate the models cost
        ## To make the math work out

        ## Calculating cross entropy
        ## Ok, was originally cross entropy but I changed it to
        ## Make the math easier on my end
        cross_entropy = .5 * (self.forward(matrix) - expected)^2

        sum_weight_ih_squares = np.square(weights_ih).sum()
        sum_weight_ho_squares = np.square(weights_ho).sum()

        ## This will encourage smaller constants in the weight matrices
        ## hopefully reducing overfitting.
        ## Will be added back eventually

        #lambda_factor = sum_weight_ho_squares  + sum_weight_ih_squares
        cost = cross_entropy #+ self.reg_lambda * lambda_factor

        return cost

    def costPrime(self, matrix, expected):
        cross_entropy_prime = self.forward(matrix) - expected.transpose()

        return cross_entropy_prime

    def getOutput(self):
        return self.output

    def displayLayers(self):
        ## Debugging tool
        print "=================="

        print "input"
        print self.input

        print "weights ih"
        print self.weights_ih

        print "hidden layer"
        print self.hidden

        print "weights ho"
        print self.weights_ho

        print "output"
        print self.output
