import os
import datetime
from Neural_networks import NN
from reader import MarksHdf5Reader
import numpy as np


'''
Here I am just testing the class's functionality and improving the output.
'''

os.sys.path.append("C:/CCM_git/marks_hdf5/")

directory = "C:/CCM_git/marks_hdf5/files/"
hdf5_name = directory + '2019_06_10_marks.hdf5'
data = MarksHdf5Reader(hdf5_name)
vol_data = data.read_vol()

x = np.array(vol_data[0])
y = np.array(vol_data[1])
y_test = y[:, 0]
y_test = y_test.reshape((24, 1))
print("Here are the values of x.shape, y.shape, and y_test.shape: \n", x.shape, y_test.shape, y_test.shape)
lr = 0.01
af = 'tanh'
num_iter = 750
num_units = [5, 4, 3, 2, 1]
num_units = np.flip(np.arange(1, 11, 1))
print("\nHere is number of units nd.array: ", num_units)

test_NN = NN(x, y_test, num_units, af, lr, num_iter)
print(test_NN)

param = test_NN.initialize_parameters_deep()

param_dim = len(param)//2 + 1
for layer in range(1, param_dim):
    print("Here are dimensions of W" + str(layer) + "'s shape: \n", param["W" + str(layer)].shape)
    print("\nHere are the dimensions of b" + str(layer), "'s shape: \n", param["b" + str(layer)].shape)


testing_L_model_forward = test_NN.L_model_forward(param)
testing_L_model_forward
print(testing_L_model_forward)

print(y_test.shape)
print("\n\n\n TESTING THE ENTIRE NEURAL NETWORK MODEL!!!!!!!!")

test_fit = test_NN.run_neural_network_classification_model()
print(type(test_fit), type(test_fit[0]), type(test_fit[1]), len(test_fit[0]), len(test_fit[1]), "Now the last 10 costs: \n", test_fit[1][90:99])

print("\nHere is parameter W1 and shape: \n")
print(test_fit[0]["W1"], test_fit[0]["W1"].shape, "\nHere is parameter W8 and shape: \n", test_fit[0]["W8"], test_fit[0]["W8"].shape)
print(test_fit[2])

neural_network_2 = NN(x, y[:, 1], num_units, af, lr, num_iter)
test_fit_2 = neural_network_2.run_neural_network_classification_model()
print(test_fit_2[0]["W1"].shape, "\nNow the costs: \n", test_fit_2[1][90:99])

neural_network_3 = NN(x, y[:, 2], num_units, af, lr, num_iter)
test_fit_with_regularization = neural_network_3.run_neural_network_classification_model(lambd=1)
