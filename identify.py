class NeuralNetwork:
    """Neural network class that allows for simple foreward and backward
    propogation with a momentum-based stochastic gradient descent method.

    Has a dependency on numpy"""

    def __init__(self, input_size, hidden_size, output_size, reg_lambda=.01):
        print "Neural network initialized with random weights"
        self.weights_ih = np.random.rand(input_size , hidden_size)
        self.weights_ho = np.random.rand(hidden_size, output_size)
        self.bias_ih    = np.random.rand(hidden_size)
        self.bias_ho    = np.random.rand(output_size)
        self.reg_lambda = reg_lambda;
        self.input      = np.zeros(input_size)
        self.hidden     = np.zeros(input_size)
        self.output     = np.zeros(input_size)

    def forwardPropogate(matrix):
        self.hidden = weights_ih.dot(self.input)  + self.bias_ih
        self.hidden = np.tanh(self.hidden)
        self.output = weights_ih.dot(self.hidden) + self.bias_ho

        return self.output

    def train(matrix, expected):
        pass

    def cost(matrix, expected):
        ## Uses cross entropy with a regularization constant added on top to
        ## Calculate the models cost

        matrix = np.tranpose(matrix)

        log_results = np.log(self.forwardPropogate(matrix))
        cross_entropy = -expected.dot(log_results)

        sum_weight_ih_squares = np.square(weights_ih).sum()
        sum_weight_ho_squares = np.square(weights_ho).sum()

        ## This will encourage smaller constants in the weight matrices
        ## hopefully reducing overfitting.

        lambda_factor = sum_weight_ho_squares  + sum_weight_ih_squares
        cost = cross_entropy + self.reg_lambda * lambda_factor
        return cost

    def getOutput():
        return self.output
