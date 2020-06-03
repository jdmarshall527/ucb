import numpy as np
import seaborn as sns


class NN:
    def __init__(self, predictor, predicted, size_of_hidden_layers, hidden_layer_activation_function, learning_rate, num_iterations):    # size_of_hidden_layers = list with number of neurons in each layer: i.e --> [5,4,3]

        if predictor.ndim == 2:
            self.X = np.transpose(predictor)                            # Applying transpose because I assume X.shape will be (number of examples, number of features)
            self.y = np.transpose(predicted)
        elif predictor.ndim == 3:
            self.X = np.transpose(predictor, (1,0,2))                   # For recurrent neural network, if we have array of (num_feat, num_examples, num time_steps)
            self.y = np.transpose(predicted, (1,0,2))                   # then we will need to transpose just the first two axes so it is of same style as two dim iteration
        self.n_h = size_of_hidden_layers
        self.layers = len(size_of_hidden_layers)
        self.activate_func = hidden_layer_activation_function
        self.lr = learning_rate
        self.num_iters = num_iterations

    def two_layer_initial_parametrization(self, n_x, n_h, n_y):
        # n_x is size of input layer x, n_h is size of hidden layer, n_y is size of output layer
        n_h = self.n_h
        n_x = self.n_x
        n_y = self.y.shape[0]
        W1 = np.random.randn(n_h, n_x) * 0.01         # initialize a random weight matrix so that values don't get stuck
        b1 = np.zeros((n_h, 1))                       # not necessary for bias terms
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        assert(W1.shape == (n_h, n_x))
        assert(b1.shape == (n_h, 1))
        assert(W2.shape == (n_y, n_h))
        assert(b2.shape == (n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}
        return parameters

    def initialize_parameters_deep(self):
        X = self.X
        layer_dims = self.n_h
        if type(layer_dims) == np.ndarray:
            layer_dims = layer_dims.tolist()
        parameters = {}
        n_x = X.shape[0]
        layer_dims.insert(0, n_x)
        L = len(layer_dims)
        # initialize every W weight matrix and bias term for each layer, l, in neural net
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*(np.sqrt(2/layer_dims[l-1]))
            parameters['b' + str(l)] = np.zeros((layer_dims[l],1))*np.sqrt(2/layer_dims[l-1])

            assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
            assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))
        # making sure there are no errors in the dimension size
        return parameters

    def linear_forward(self, A, W, b):
        # print("From linear_forward: \n", "W's shape: \n", W.shape, "A's shape: \n", A.shape, "b's shape: \n", b.shape)
        Z = np.dot(W, A) + b

        assert(Z.shape == (W.shape[0], A.shape[1]))
        cache = (A, W, b)                         # store the activations, weights, and biases for backward prop

        return Z, cache                           # return the regression to pass to activation func (sigmoid/tanh/reLU)

    def linear_activation_forward(self, A_prev, W, b, activation_function):
        Z = 0
        A = 0
        if activation_function == 'sigmoid':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = 1/(1 + np.exp(-Z))
        elif activation_function == 'tanh':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = np.tanh(Z)
        elif activation_function == 'relu':
            Z, linear_cache = self.linear_forward(A_prev, W, b)
            A = max(0, Z)
        else:
            print("Bad string for activation function: select 'sigmoid', 'tanh', or 'relu'")

        assert(A.shape == (W.shape[0], A_prev.shape[1]))

        activation_cache = Z
        cache = (linear_cache, activation_cache)

        return A, cache

    def L_model_forward(self, parameters):
        X = self.X
        base_activation_func = self.activate_func
        iter_length = len(parameters)//2         # two params for each iteration: W_n, b_n for layer n, so divide by two
        caches = []                              # create a meta-cache
        A = X

        # Now iterating through each layer and calculating the activation value,
        # and storing the resulting cache in our list of cache
        # Since for loops iterate from 0 until length -1, I am using a
        # relu or tanh activation function for the first L layers to
        # mitigate vanishing gradient problem that we have with sigmoid func

        for layer in range(1, iter_length):
            # print("Here is the iterations in L_model_forward I've made it through: \n", layer)
            A_prev = A
            if base_activation_func == 'relu':
                A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(layer)], parameters['b' + str(layer)], base_activation_func)
                caches.append(cache)
            elif base_activation_func == 'tanh':
                A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(layer)], parameters['b' + str(layer)], base_activation_func)
                caches.append(cache)
            else:
                print("Bad activation name: pick tanh, or relu")

        AL, cache = self.linear_activation_forward(A, parameters['W' + str(iter_length)], parameters['b' + str(iter_length)], 'sigmoid')

        # AL is activation of layer L, last layer
        caches.append(cache)

        return AL, caches

    def compute_cost(self, AL, lambd=0):  # computing cross entropy cost for logit regress sigmoid
        Y = self.y
        num_examples = Y.shape[0]
        layers = int(len(param)//2)
        param_squared_list = []

        cost = (-1/num_examples)*np.sum((Y*np.log(AL) + (1-Y)*np.log(1-AL)))    # J for cost function

        # To add regularization, we need to take each parameter estimate W_n (for layer n)
        # and square each weight element wise, and then sum every weight in W_n
        # in below, I am iterating through the dictionary param, taking each weight matrix
        # and element wise squaring and putting into a list so np.sum works in the aggregate sum

        for i in range(layers):
            parameter = np.square(param["W" + str(i+1)])
            parameter_summed = np.sum(parameter)
            param_squared_list.append(parameter_summed)

        # Once we have our dictionary of only W_n element wise squared, summed, and converted into a list, we use np.sum

        cost = cost + (lambd/(2*num_examples))*np.sum(param_squared_list)

        cost = np.squeeze(cost)                        # This step just in case its returned as a 1D array into a number
        assert(cost.shape == ())                       # This ensures that its functioning properly
        return cost

    def linear_backward(self, dZ, linear_cache, lambd):

        A_prev, W, b = linear_cache
        num_examples = A_prev.shape[1]
        del_W = np.dot(dZ, A_prev.T)/float(num_examples) + (lambd/num_examples)*W
        del_b = np.sum(dZ, axis=1, keepdims=True)/float(num_examples)
        del_A_prev = np.dot(W.T, dZ)

        assert(del_A_prev.shape == A_prev.shape)
        assert(del_W.shape == W.shape)
        assert(del_b.shape == b.shape)

        return del_A_prev, del_W, del_b

    def sigmoid_derivative(self, Z):

        d_dZ = np.exp(-Z)/np.square(1 + np.exp(-Z))
        return d_dZ

    def sigmoid_backward(self, del_A_prev, Z):

        del_Z = np.multiply(del_A_prev, self.sigmoid_derivative(Z))
        return del_Z

    def relu_backward(self, del_A_prev, Z):
        Z[Z < 0] = 0
        del_A_times_Z = del_A_prev * Z
        return del_A_times_Z

    def tanh_derivative(self, Z):
        del_tanh = 1 - np.square(np.tanh(Z))
        return del_tanh

    def tanh_backward(self, del_A_prev, Z):
        d_dZ = np.multiply(del_A_prev, self.tanh_derivative(Z))
        return d_dZ

    def linear_activation_backward(self, dA, cache, activation_function, lambd):

        linear_cache, activation_cache = cache

        if activation_function == 'relu':
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, lambd)
        elif activation_function == 'tanh':
            dZ = self.tanh_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, lambd)
        elif activation_function == 'sigmoid':
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache, lambd)

        return dA_prev, dW, db

    def L_model_backward(self, AL, Y, caches, lambd):
        Y = self.y
        # print("\nIN L_MODEL_BACKWARD: Here is Y.shape, and AL.shape:\n", Y.shape, AL.shape)
        grads = {}
        L = len(caches)                                     # gives you the number of layers
        m = AL.shape[1]
        Y = Y.reshape(AL.shape)

        # ultimately, you will have A1 = sigmoid(Z1=W1.X+b1) then A2 = sigmoid(Z2 = W2.A1+b2) ...etc
        # but your final shape of the output layer should be identical to the shape of the y input layer

        dAL = - (np.divide(Y, AL) - np.divide((1-Y), (1-AL)))

        current_cache = self.linear_activation_backward(dAL, caches[L-1], 'sigmoid', lambd)
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = current_cache

        for l in reversed(range(L-1)):
            if self.activate_func == 'relu':
                current_cache = self.linear_activation_backward(grads["dA" + str(l+1)], caches[l], 'relu', lambd)
            elif self.activate_func == 'tanh':
                current_cache = self.linear_activation_backward(grads["dA" + str(l+1)], caches[l], 'tanh', lambd)
            dA_prev_temp, dW_temp, db_temp = current_cache
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l+1)] = dW_temp
            grads["db" + str(l+1)] = db_temp

        return grads

    def update_parameters(self, parameters, grads, lambd):
        learning_rate = self.lr
        L = len(parameters) // 2
        num_examples = self.X.shape[1]

        for l in range(L):
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*(lambd/num_examples)*grads["dW" + str(l+1)]
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]

        return parameters

    # this takes an X of size (num_example, num_features) and a y of size (num_examples, 1) (typically a binary coding)
    def run_neural_network_classification_model(self, lambd=0):

        predictor = self.X
        predicted = self.y
        num_hidden_units = self.n_h
        num_iterations = self.num_iters

        cost_print_array = np.arange(1, 11, 1)*num_iterations/10     # used to define even intervals to print cost

        if type(num_hidden_units) == np.ndarray:
            num_hidden_units = num_hidden_units.tolist()

        n_x = predictor.shape[0]                                 # predictor data should be [r,c] = [features, examples]
        n_y = predicted.shape[0]                                 # predicted data should be [r,c] = [outcome, examples]
        num_layers = len(num_hidden_units) + 1
        num_iterations = int(num_iterations)

        if num_layers == 2:
            parameters = self.two_layer_initial_parametrization(n_x, n_y, num_hidden_units)
        else:
            parameters = self.initialize_parameters_deep()

        cost_recorded = []

        for iterations in range(0, num_iterations):
            AL, caches = self.L_model_forward(parameters)
            # returns: AL, caches  ---> caches is a meta-caches of linear_caches and activation_caches,
            # linear_caches have: (A,W,b), activation_caches have: (A) --- these are all lists
            cost = self.compute_cost(AL, lambd)
            cost_recorded.append(cost)
            grads = self.L_model_backward(AL, predicted, caches, lambd)
            parameters = self.update_parameters(parameters, grads, lambd)

            if iterations in cost_print_array:
                print("\n Iteration number:", iterations)
                print("\n Cost:", cost)

        sns.set_style('darkgrid')

        plt.plot(cost_recorded)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Cost')
        plt.title('Neural Network Fit')
        plt.show()

        y_predicted = AL
        difference = y_predicted - self.y
        output = (parameters, cost_recorded, AL, difference)
        plt.scatter(AL, self.y)
        plt.xlabel('Predicted')
        plt.ylabel('Observed')
        plt.title('Predicted vs Observed')
        plt.show()


        sns.distplot(np.transpose(AL), bins=5)
        plt.title("Histogram and Density of Predictions")
        plt.xlabel("Predicted Values")
        plt.ylabel("Density")


        plt.show()


        return output

    ### Now implementing Recurrent Neural Network in this class

    def rnn_initialization(self):
        num_examples = self.X.shape[1]
        num_features = self.X.shape[0]
        num_timesteps = self.X.shape[2]
        parameters = {}

        a_zero = np.zeros((num_features, num_examples, num_timesteps))
        parameters["Waa"] = np.random.randn((self.n_h, num_features, num_timesteps))
        parameters["Wax"] = np.random.randn((self.n_h, num_features, num_timesteps))
        parameters["ba"] = np.zeros((num_features, 1, num_timesteps))

        parameters["Wya"] = np.random.randn((num_features, self.n_h, num_timesteps))
        parameters["by"] = np.zeros((num_features, 1, num_timesteps))

        return parameters, a_zero

    # This function basically takes an numpy array and outputs a probability vector
    def softmax_function(self, vectorized_input):
        divisor = np.sum(np.exp(vectorized_input))
        probability_vector = np.exp(vectorized_input)/divisor
        return probability_vector

    # This function calculates on instance of a rnn forward propagation where
    # a_t = tanh(Wax * x_t + Waa*a(t-1) + ba) and
    # yhat_t = softmax(Wya * a_t + by)
    # Basically, you are running a neural network for every observation of the sequential data
    def rnn_cell_forward(self, xt, a_prev, parameters):
        Wax = parameters["Wax"]
        Waa = parameters["Waa"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]

        if self.activate_func == "tanh":
            a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
        elif self.activate_func == "sigmoid":
            a_next = 1/(1 + np.exp(-1* (np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)))
        elif self.activate_func == "relu":
            a_next = max(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)
        else: print("Bad activation function!")

        yt_pred = np.dot(Wya, a_next) + by
        #yt_pred = self.softmax_function(np.dot(Wya, a_next) + by)  --> this is only for classification

        cache = (a_next, a_prev, xt, parameters)
        return a_next, yt_pred, cache

    def rnn_forward(self, x_t, a0, parameters):

        """
        Arguments:
        x -- Input data for every time-step, of shape (n_x, m, T_x).
        a0 -- Initial hidden state, of shape (n_a, m)
        parameters -- python dictionary containing:
                            Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                            Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                            Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                            ba --  Bias numpy array of shape (n_a, 1)
                            by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

        Returns:
        a -- Hidden states for every time-step, numpy array of shape (n_a, m, T_x)
        y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
        caches -- tuple of values needed for the backward pass, contains (list of caches, x)
        """

        caches = []
        n_x, m, T_x = x_t.shape
        n_y, n_a = parameters["Wya"].shape
        a = np.zeros([n_a, m, T_x])
        y_pred = np.zeros([n_y, m, T_x])

        a_next = a0
        for t in range(T_x):
            xt = x[:, :, t]
            a_next, yt_pred, cache = self.rnn_cell_forward(xt, a_next, parameters)
            a[:, :, t] = a_next
            y_pred[:, :, t] = yt_pred
            caches.append(cache)

        caches = (caches, x_t)

        return a, y_pred, caches

    def compute_least_squares_cost(self, A_t, y_t, param, lambd=0):

        L = len(param)//2
        num_examples = y_t.shape[1]
        cost = (-1/num_examples)*np.sum(np.square(np.subtract(A_t, y_t)))
        param_list = []

        for i in range(L):
            parameter = param["W" + str(i)]
            parameter_squared = np.square(parameter)
            parameter_squared_summed = np.sum(parameter_squared)
            param_list.append(parameter_squared_summed)

        reg_cost = (lambd/(2*num_examples))*np.sum(param_list)

        total_cost = cost + reg_cost

        return total_cost

    # Note rnn_cell_backward does not include the calculation of loss from  y⟨t⟩y⟨t⟩ , this is incorporated into the incoming da_next. This is a slight mismatch with rnn_cell_forward which includes a dense layer and softmax.

    def rnn_cell_backward(self, da_next, cache):
        (a_next, a_prev, xt, parameters) = cache
        Wax = parameters["Wax"]
        Waa = parameters["Waa"]
        Wya = parameters["Wya"]
        ba = parameters["ba"]
        by = parameters["by"]

        dz = 1 - np.square(np.tanh(np.dot(Wax, xt) + np.dot(Waa, a_prev) + ba))
        dxt = np.dot(np.transpose(Wax), np.multiply(da_next, dz))
        dWax = np.dot(np.multiply(da_next, dz), np.transpose(xt))
        da_prev = np.dot(np.transpose(Waa), np.multiply(da_next, dz))
        dWaa = np.dot(np.multiply(da_next, dz), np.transpose(da_prev))
        dba = np.dot(np.multiply(da_next, dz), np.transpose(np.ones((1, da_prev.shape[1]))))

        gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

        return gradients

    def rnn_backward(self, da, caches):

        (caches, x) = caches[0], da
        (a1, a0, x1, parameters) = caches[1]
        n_a, m, T_x = a0.shape
        n_x, m = x1.shape
        dx = np.zeros(a0.shape)
        dWax = np.zeros(n_a, n_x)
        dWaa = np.zeros(n_a, n_a)
        dba = np.zeros (n_a, 1)
        da0 = np.zeros(n_a, m)
        da_prevt = np.zeros(n_a, m)

        for t in reversed(len(caches[2])):
            gradients = self.rnn_cell_backward(da, caches[::t])
            dxt, da_prevt, dWaxt, dWaat, dbat = gradients["dxt"], gradients["da_prev"], gradients["dWax"], gradients["dWaa"], gradients["dba"]
            dx[:, :, t] = dxt + da0
            dWax += dWax
            dWaa += dWaa
            dba += dba

        da0 = da_prevt

        gradients = {"dx": dx, "da0": da0, "dWax": dWax, "dWaa": dWaa, "dba": dba}

        return gradients

    def run_rnn_model(self, lambd):
        x_t = self.X
        y_t = self.y
        num_timesteps = x_t.shape[2]            #3 dim array (num feat, num examples (probably observations per day), number of days)

        parameters, a0 = self.rnn_initialization(self)

        model = self.rnn_forward(x_t, a0, parameters)
