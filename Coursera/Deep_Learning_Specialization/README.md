# Neural Network Class

In this specialization, the curriculum helps you build your own neural network from scratch.  I filled in some blanks and started adding some additional functionality. I still have some more work to do, but wanted to get some of this uploaded and some thoughts jotted down in case I don't remember when I return to it.

**Content**

This class takes in an array of inputs (x, y, layers, activation function, learning rate, number of iterations):

1. x should be np.array (number of features, number of examples)
2. y is what x is trying to predict, same format as above
3. Layers is how large the inner layers of the neural network should be.
    1. Should be an np.array.
        1. Something like np.flip(np.arange(1,11,1)) creates an array [10,9,...,3,2,1] --> These integers represent the number of               neurons at each layer of the neural network.
4. Activation function is the activation function for the *inner layers*, the output layer uses a sigmoid activation function.
    1. Activation function one of these three strings: 'relu', 'sigmoid' or 'tanh'
5. The learning rate is the size of the step opposite the partial derivative for that iteration.  Larger learning rate means it might take fewer iterations to converge but it might also diverge, this should be a float.
6. Number of iterations is number of times to repeat the process, this should be an integer.

Once you initialize an NN object, you can call nn_object.run_neural_network_classification_model() this will fit the model for you, printing out the cost of the fit every time it completes 10% of the total number of iterations, and then produce the charts that I attached in this folder.  The scratch file shows how I tested this model along with the seaborn charts that are generated when you call the model.  I also included the console output from the scratch file just to be transparent and comprehensive.
