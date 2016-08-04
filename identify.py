class NeuralNetwork:
    """Neural network class that allows for simple foreward and backward
    propogation with a momentum-based stochastic gradient descent method.

    Has a dependency on numpy"""

    def __init__(self, input_size, hidden_size, output_size,
                 learn_rate = .1, reg_lambda=.01):
        print "Neural network initialized with random weights"
        self.weights_ih = np.random.rand(input_size , hidden_size)
        self.weights_ho = np.random.rand(hidden_size, output_size)
        self.bias_ih    = np.random.rand(hidden_size)
        self.bias_ho    = np.random.rand(output_size)
        self.learn_rate = learn_rate
        self.reg_lambda = reg_lambda
        self.input      = np.zeros(input_size)
        self.hidden_lin = np.zeros(input_size)
        self.hidden     = np.zeros(input_size)
        self.output     = np.zeros(input_size)

    def forward(matrix):
        ## Propegating results foreward and applying a nonlinearity
        self.hidden_lin = weights_ih.dot(self.input)  + self.bias_ih
        self.hidden     = self.act(self.hidden_lin)
        self.output = weights_ih.dot(self.hidden) + self.bias_ho

        ## Softmaxing output
        self.output = np.exp(self.output)
        return self.output/np.sum(self.output)

    def train(self, matrix, expected):
        current_cost_prime = self.costPrime(matrix, expected)

        ## Calculates the layer 3 error
        error_layer_3 = current_cost_prime.multiply(self.output)
        dJdW_ho = self.hidden.tranpose().dot(error_layer_3)

        ## Calculates the error for layer two by propogating back the error
        ## And applying the chain rule
        error_layer_2 = error_layer_3 * self.weights_ho.transpose()
                                      * self.actPrime(hidden_lin)
        dJdW_ih = self.matrix.tranpose().dot(error_layer_2)

        weights_ho = weights_ho - dJdW_ho * self.learn_rate
        weights_ih = weights_ih - dJdW_ih * self.learn_rate

    def act(matrix):
        ## Activation function
        return np.tanh(matrix)

    def actPrime(matrix):
        ## Derivative of activation function
        return np.square(np.sech(matrix))

    def cost(matrix, expected):
        ## Uses cross entropy with a regularization constant added on top to
        ## Calculate the models cost
        ## To make the math work out
        matrix = np.transpose(matrix)

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

    def costPrime(matrix, expected):
        matrix = np.transpose(matrix)
        cross_entropy_prime = self.forward(matrix) - expected

        return cross_entropy_prime

    def getOutput():
        return self.output
